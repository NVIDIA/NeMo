    @measure_eta 
    def perform_streaming_stt_spk(
        self,
        step_num,
        chunk_audio,
        chunk_lengths,
        cache_last_channel,
        cache_last_time,
        cache_last_channel_len,
        previous_hypotheses,
        asr_pred_out_stream,
        diar_pred_out_stream,
        streaming_state,
        # mem_last_time,
        # fifo_last_time,
        left_offset,
        right_offset,
        is_buffer_empty,
        pad_and_drop_preencoded,
    ):

        (
            asr_pred_out_stream,
            transcribed_texts,
            cache_last_channel,
            cache_last_time,
            cache_last_channel_len,
            previous_hypotheses,
        ) = self.asr_model.conformer_stream_step(
            processed_signal=chunk_audio,
            processed_signal_length=chunk_lengths,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=is_buffer_empty,
            previous_hypotheses=previous_hypotheses,
            previous_pred_out=asr_pred_out_stream,
            drop_extra_pre_encoded=calc_drop_extra_pre_encoded(
                self.asr_model, step_num, pad_and_drop_preencoded
            ),
            return_transcription=True,
        )

        if step_num > 0:
            left_offset = 8
            chunk_audio = chunk_audio[..., 1:]
            chunk_lengths -= 1
        # (
        #     mem_last_time,
        #     fifo_last_time,
        #     mem_preds,
        #     fifo_preds,
        #     diar_pred_out_stream
        # ) = self.diar_model.forward_streaming_step(
        #     processed_signal=chunk_audio.transpose(1, 2),
        #     processed_signal_length=chunk_lengths,
        #     spkcache_last_time=mem_last_time,
        #     fifo_last_time=fifo_last_time,
        #     previous_pred_out=diar_pred_out_stream,
        #     left_offset=left_offset,
        #     right_offset=right_offset,
        # )
            streaming_state, diar_pred_out_stream = self.diar_model.forward_streaming_step(
            processed_signal=chunk_audio.transpose(1, 2),
            processed_signal_length=chunk_lengths,
            streaming_state=streaming_state,
            total_preds=diar_pred_out_stream,
            left_offset=left_offset,
            right_offset=right_offset,
            )
        # if step_num > 30:
        #     import ipdb; ipdb.set_trace()
        transcribed_speaker_texts = [None] * len(self.test_manifest_dict)
        for idx, (uniq_id, data_dict) in enumerate(self.test_manifest_dict.items()): 
            if not (len( previous_hypotheses[idx].text) == 0 and step_num <= self._initial_steps):
                # Get the word-level dictionaries for each word in the chunk
                self._word_and_ts_seq[idx] = self.get_frame_and_words_online(uniq_id=uniq_id,
                                                                            step_num=step_num, 
                                                                            diar_pred_out_stream=diar_pred_out_stream[idx, :, :],
                                                                            previous_hypothesis=previous_hypotheses[idx], 
                                                                            word_and_ts_seq=self._word_and_ts_seq[idx],
                                                                            )
                if len(self._word_and_ts_seq[idx]["words"]) > 0:
                    self._word_and_ts_seq[idx] = self.get_sentences_values(session_trans_dict=self._word_and_ts_seq[idx], 
                                                                           sentence_render_length=self._sentence_render_length)
                    if self.cfg.generate_scripts:
                        transcribed_speaker_texts[idx] = \
                            print_sentences(sentences=self._word_and_ts_seq[idx]["sentences"], 
                            color_palette=get_color_palette(), 
                            params=self.cfg)
                        write_txt(f'{self.cfg.print_path}'.replace(".sh", f"_{idx}.sh"), 
                                  transcribed_speaker_texts[idx].strip())
            
        return (transcribed_speaker_texts,
                transcribed_texts,
                asr_pred_out_stream,
                transcribed_texts,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
                previous_hypotheses,
                streaming_state,
                # mem_last_time,
                # fifo_last_time,
                diar_pred_out_stream)